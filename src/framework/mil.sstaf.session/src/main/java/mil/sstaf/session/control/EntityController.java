/*
 * Copyright (c) 2022
 * United States Government as represented by the U.S. Army DEVCOM Analysis Center.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mil.sstaf.session.control;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.SuperBuilder;
import lombok.extern.jackson.Jacksonized;
import mil.sstaf.blackboard.api.AddEntryRequest;
import mil.sstaf.core.entity.*;
import mil.sstaf.core.features.HandlerContent;
import mil.sstaf.core.json.JsonLoader;
import mil.sstaf.core.util.Injector;
import mil.sstaf.core.util.RNGUtilities;
import mil.sstaf.core.util.SSTAFException;
import mil.sstaf.session.messages.Error;
import mil.sstaf.session.messages.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;


/**
 * Central coordinator for processing Events and dispatching messages.
 */
@SuperBuilder
@Jacksonized
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, property = "class")
public final class EntityController extends BaseEntity {

    public static final String SYSTEM_ENTITY_CONTROLLER = "SYSTEM:EntityController";

    private static final Logger logger = LoggerFactory.getLogger(EntityController.class);

    @Getter
    private final Map<Force, List<BaseEntity>> entities;

    @Getter
    private final int executorThreads;
    //
    // Executor
    //
    @Builder.Default
    private Collection<RunAgentsCallable> runAgentsTasks = null;
    @Builder.Default
    private Collection<ProcessEventsCallable> processEventsTasks = null;
    @Builder.Default
    private ExecutorService executorService = null;

    //
    // EntityRegistry
    //
    @Builder.Default
    private EntityRegistry registry = null;

    //
    // ClientProxy - used for interaction with the Session. Enables
    // Entity behavior without exposing Entities through the Session.
    //
    @Builder.Default
    private ClientProxy clientProxy = null;

    @Getter
    @Builder.Default
    private long lastTickTime_ms = 0;

    @Getter
    @Builder.Default
    private long nextEventTime_ms = 0;

    /**
     * Constructor
     *
     * @param builder the builder that was generated by Lombok
     */
    private EntityController(EntityControllerBuilder<?, ?> builder) {
        super(builder);
        // Replace the user-space id with a system-space id.
        this.executorThreads = builder.executorThreads == 0 ? Runtime.getRuntime().availableProcessors() : builder.executorThreads;
        this.id = BlockCounter.systemCounter.getID();
        this.clientProxy = ClientProxy.builder().build();
        this.entities = builder.entities;
        this.handle.setForce(Force.SYSTEM);
        this.runAgentsTasks = new ArrayList<>();
        this.processEventsTasks = new ArrayList<>();

        this.registry = new EntityRegistry();
        this.registry.setClientAddress(clientProxy.getHandle());
        this.registry.registerEntities(entities);
        this.registry.registerEntity(Force.SYSTEM, this);
        this.registry.registerEntity(Force.SYSTEM, clientProxy);

        this.registry.compileEntityMaps();

        this.registry.getSimulationEntities().forEach(entity -> {
            prepareEntity(entity);
            runAgentsTasks.add(new RunAgentsCallable(entity));
            processEventsTasks.add(new ProcessEventsCallable(entity));
        });

        this.clientProxy.setForce(Force.SYSTEM);
        this.clientProxy.setName("ClientProxy");
        this.clientProxy.setRegistry(registry);
        this.clientProxy.init();
        /*
         * Features running in the EntityController probably need access to the registry.
         */
        this.featureManager.injectAll(registry);
        if (this.executorThreads > 1) {
            this.executorService = Executors.newFixedThreadPool(this.executorThreads);
        } else {
            this.executorService = Executors.newSingleThreadExecutor();
        }

        //
        // Create and register a Handler to process messages addressed
        // to the EntityController.
        //
        EntityControllerHandler ech = new EntityControllerHandler(this);
        this.featureManager.register(ech);
        init();
    }

    public static EntityController from(File file) {
        Path p = Path.of(file.getPath());
        JsonLoader jsonLoader = new JsonLoader();
        return jsonLoader.load(p, EntityController.class);
    }

    public void shutdown() {
        this.executorService.shutdown();
    }

    /**
     * Performs final preparations on the Entity
     *
     * @param entity the Entity to prepare
     */
    private void prepareEntity(Entity entity) {
        if (logger.isInfoEnabled()) {
            logger.info("Configuring {}", entity.getName());
        }

        //
        // Need to set the seed for each entity here before init()ing
        //
        long subSeed = RNGUtilities.generateSubSeed(randomGenerator);
        logger.info("Setting seed in {} to {}", entity.getName(), subSeed);
        Injector.inject(entity, "randomSeed", subSeed);
        entity.injectInFeatures(registry);
        //
        // Initialize it!
        //
        entity.init();

        //
        // Register this controller with each Entity using a message.
        //
        if (entity.canHandle(AddEntryRequest.class)) {
            HandlerContent content = AddEntryRequest.from(handle);
            EntityAction entityAction = EntityAction.builder()
                    .destination(Address.makeExternalAddress(entity.getHandle()))
                    .source(Address.makeExternalAddress(getHandle()))
                    .sequenceNumber(generateSequenceNumber())
                    .respondTo(Address.NOWHERE).content(content).build();
            entity.receive(entityAction);
        }

        //
        // Additional messages?
        //

        //
        // Process messages
        //
        entity.processMessages(Long.MIN_VALUE);
        if (logger.isInfoEnabled()) {
            logger.info("Done configuring {}", entity.getName());
        }
    }

    /**
     * Accepts a {@code BaseSessionCommand} from the client and routes it for processing.
     *
     * @param command the command
     */
    public void submitCommand(final Command command) {
        clientProxy.submitCommand(command);
    }

    /**
     * Accepts a {@code SSTAFEvent} from the client and routes it for processing.
     *
     * @param event the event
     */
    public void submitEvent(final Event event) {
        clientProxy.submitEvent(event);
    }

    /**
     * Gets outbound Messages
     *
     * @return SSTAFResults generated from outbound internal messages
     */
    public List<BaseSessionResult> getMessagesToSession() {
        List<Message> messages = clientProxy.takeInbound();
        List<BaseSessionResult> output = new ArrayList<>();
        messages.forEach(message -> {
            if (message instanceof MessageResponse) {
                MessageResponse mr = (MessageResponse) message;
                BaseSessionResult sstafResult = convertMessageToResult(mr);
                output.add(sstafResult);
            }
        });
        return output;
    }

    /**
     * Returns the depth of the Session Proxy's queue
     *
     * @return the queue depth.
     */
    public int getSessionProxyQueueDepth() {
        return clientProxy.getQueueDepth();
    }

    /**
     * Processes all the actions and events up to the specified simulation time.
     * <p>
     * This is the central method in the Soldier and Squad Trade-space Analysis Framework.
     *
     * @param currentTime_ms the current simulation time.
     * @return the time of the next added event
     */
    public SessionTickResult tick(long currentTime_ms) {
        logger.debug("Executing tick at {}", currentTime_ms);

        List<Future<Long>> nextTimes1;
        lastTickTime_ms = currentTime_ms;
        try {
            runAgentsTasks.forEach(task -> task.setCurrentTime(currentTime_ms));
            nextTimes1 = executorService.invokeAll(runAgentsTasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
            nextTimes1 = List.of();
        }

        routeMessages();

        //
        // Entity controller runs in reverse. First messages are received and processed,
        // then Agents wrap up global tasks and push outcomes and global state
        //
        this.processMessages(currentTime_ms);
        this.runAgents(currentTime_ms);
        routeMessages();

        List<Future<Long>> nextTimes2;
        try {
            processEventsTasks.forEach(task -> task.setCurrentTime_ms(currentTime_ms));
            nextTimes2 = executorService.invokeAll(processEventsTasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
            nextTimes2 = List.of();
        }

        routeMessages();

        List<BaseSessionResult> toSession = getMessagesToSession();
        long nextEventTime_ms = Long.min(getMinTime(nextTimes1), getMinTime(nextTimes2));

        return SessionTickResult.builder().nextEventTime_ms(nextEventTime_ms)
                .messagesToClient(toSession)
                .build();
    }

    /**
     * Ticks the simulation again using the last tick time.
     * <p>
     * Commands and queries are only executed when a tick happens. Consequently,
     * to make a query or issue a command without changing the state of the
     * simulation the approach is to invoke tick again at the same time. Features
     * should note that the current time is the same and that no state-changing
     * actions should be taken.
     * <p>
     * TODO: A Feature compliance test needs to be developed that ensures that commands and queries are idempotent if the time is unchanged.
     *
     * @return a {@code SessionTickResult} containing any results from the tick.
     */
    public SessionTickResult tickAgain() {
        return tick(lastTickTime_ms);
    }


    /**
     * Determines the time of the next event after all events have been processed.
     *
     * @param times the futures for the processing tasks
     * @return the minimum time
     */
    private long getMinTime(List<Future<Long>> times) {
        long minTime_ms = Long.MAX_VALUE;
        logger.info("times = {}", times);
        for (Future<Long> fd : times) {
            logger.info("fd = {}", fd);
            try {
                long nt = fd.get();
                minTime_ms = Math.min(minTime_ms, nt);
            } catch (InterruptedException e) {
                logger.error("Interrupted!");
                e.printStackTrace();
            } catch (ExecutionException e) {
                logger.error("Broken! " + e.getMessage());
                e.printStackTrace();
            }
        }
        return minTime_ms;
    }

    /**
     * Resolves a path to find the EntityHandle
     *
     * @param path the entity path to look up
     * @return an {@code Optional} that contains the found {@code EntityHandle} or
     * is empty if the path was invalid.
     */
    public Optional<EntityHandle> getHandleFromPath(String path) {
        return registry.getHandle(path);
    }

    /**
     * Takes messages from a MessageDriven object and routes them to the specified
     * recipients.
     *
     * @param routeFrom the MessageDriven instance from which to task the messages.
     */
    private void routeFromMessageDriven(final Entity routeFrom) {
        List<Message> messages = routeFrom.takeOutbound();
        logger.info("Routing messages from {}, got {} messages to route", routeFrom.getName(), messages.size());
        messages.forEach(message -> {
            if (message == null) {
                logger.warn("Message is null");
            } else if (message.getDestination() == null) {
                logger.warn("Message destination is null, source = {}, content = {}", message.getSource(), message.getContent());
            } else if (message.getDestination().equals(Address.NOWHERE)) {
                logger.debug("Dropping message from {} to NOWHERE, contents = {}", message.getSource(), message.getContent().getClass());
            } else {
                logger.debug("Routing from {} to {}, contents = {}", message.getSource(), message.getDestination().entityHandle.getName(), message.getContent().getClass());
                Optional<Entity> optionalEntity = registry.getEntityByHandle(message.getDestination().entityHandle);
                optionalEntity.ifPresent(entity -> entity.receive(message));
            }
        });
    }

    /**
     * Routes all messages from all entities
     */
    private void routeMessages() {
        logger.info("Routing messages");
        var allEntities = registry.getAllEntities();
        // split for debugging
        allEntities.forEach(this::routeFromMessageDriven);
    }

    public BaseSessionResult convertMessageToResult(final MessageResponse response) {
        if (response instanceof ErrorResponse) {
            ErrorResponse errorResponse = (ErrorResponse) response;
            return convertError(errorResponse);
        } else {
            return convertSuccess(response);
        }
    }

    private Error convertError(final ErrorResponse errorResponse) {
        return Error.builder()
                .id(errorResponse.getMessageID())
                .entityPath(errorResponse.getSource().entityHandle.getForcePath())
                .throwable(errorResponse.getThrowable())
                .build();
    }

    private BaseSessionResult convertSuccess(final MessageResponse response) {
        return CommandResult.builder()
                .id(response.getMessageID())
                .entityPath(response.getSource().entityHandle.getForcePath())
                .content(response.getContent())
                .build();
    }

    public SortedSet<EntityHandle> getSimulationEntityHandles() {
        SortedSet<EntityHandle> handles = new TreeSet<>();
        registry.getSimulationEntities().forEach(entity -> handles.add(entity.getHandle()));
        return handles;
    }

    public List<String> getEntityPaths() {
        List<String> paths = new ArrayList<>();
        for (var entry : entities.entrySet()) {
            Force f = entry.getKey();
            for (var entity : entry.getValue()) {
                String fullPath = f.name() + ENTITY_PATH_DELIMITER + entity.getPath();
                paths.add(fullPath);
            }
        }
        return paths;
    }

    public String getPath() {
        return SYSTEM_ENTITY_CONTROLLER;
    }

    @SuperBuilder
    static class ClientProxy extends BaseEntity {

        @Setter
        private EntityRegistry registry;

        /**
         * Constructor
         *
         * @param builder the {@code Builder} to use to construct the {@code Entity}
         */
        protected ClientProxy(ClientProxyBuilder<?, ?> builder) {
            super(builder);
            handle.setForce(Force.SYSTEM);
        }

        @Override
        public String getPath() {
            return "SYSTEM:ClientProxy";
        }


        /**
         * Updates the command with the EntityHandle if it isn't set
         *
         * @param command the {@code BaseSessionCommand} to update.
         */
        private void resolvePath(Command command) {
            Optional<EntityHandle> optEH = registry.getHandle(command.getRecipientPath());
            optEH.ifPresentOrElse(command::setHandle,
                    () -> {
                        throw new SSTAFException("Entity Path '" + command.getRecipientPath() + "' does not exist");
                    }
            );
        }

        private void submitCommand(final Command command) {
            resolvePath(command);
            var b = EntityAction.builder()
                    .destination(Address.makeExternalAddress(command.getHandle()))
                    .source(Address.makeExternalAddress(getHandle()))
                    .content(command.getContent())
                    .respondTo(Address.makeExternalAddress(getHandle()));
            outboundQueue.offer(b.build());
        }

        public void submitEvent(final Event event) {
            resolvePath(event);
            var b = EntityEvent.builder().destination(Address.makeExternalAddress(event.getHandle())).source(Address.makeExternalAddress(getHandle())).eventTime_ms(event.getEventTime_ms()).content(event.getContent()).respondTo(Address.makeExternalAddress(getHandle()));
            outboundQueue.offer(b.build());
        }

        public int getQueueDepth() {
            return outboundQueue.size();
        }
    }

    /**
     *
     */
    private static class RunAgentsCallable implements Callable<Long> {
        final Entity entity;
        long currentTime_ms;

        RunAgentsCallable(Entity entity) {
            this.entity = entity;
        }

        void setCurrentTime(final long currentTime_ms) {
            logger.info("Updating time in RunAgentsCallable to {}", currentTime_ms);
            this.currentTime_ms = currentTime_ms;
        }

        public Long call() {
            logger.info("Invoking runAgents for {}", entity.getName());
            long res = entity.runAgents(currentTime_ms);
            logger.info("runAgents for {} returned {}", entity.getName(), res);
            return res;
        }
    }

    private static class ProcessEventsCallable implements Callable<Long> {
        final Entity entity;
        long currentTime_ms;

        ProcessEventsCallable(Entity entity) {
            this.entity = entity;
        }

        void setCurrentTime_ms(final long currentTime_ms) {
            logger.info("Updating time in ProcessEventsCallable to {}", currentTime_ms);
            this.currentTime_ms = currentTime_ms;
        }

        public Long call() {
            logger.info("Invoking processMessages for {}", entity.getName());
            long res = entity.processMessages(currentTime_ms);
            logger.debug("processMessages for {} returned {}", entity.getName(), res);
            return res;
        }
    }
}


